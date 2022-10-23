#include <charconv>
#include <future>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <pugixml.hpp>

#include "CLI11.hpp"

class Geometry
{
    pugi::xml_node m_geometry;
    static std::vector<cv::Point>
    parse_points(const pugi::xml_node &node_with_point_attr)
    {
        const auto ptr_start =
            node_with_point_attr.attribute("points").as_string();
        const auto ptr_end = ptr_start + strlen(ptr_start);
        auto cur = ptr_start;
        std::vector<cv::Point> pts;
        while (cur < ptr_end)
        {
            auto x_coord_end = strchr(cur, ',');
            if (x_coord_end == nullptr)
                x_coord_end = ptr_end;
            auto y_coord_end = strchr(cur, ';');
            y_coord_end = (y_coord_end != nullptr) ? y_coord_end : ptr_end;

            int x, y;
            std::from_chars(cur, x_coord_end, x);
            std::from_chars(x_coord_end + 1,
                            (y_coord_end == nullptr) ? ptr_end : y_coord_end,
                            y);
            cur = y_coord_end + 1;
            pts.emplace_back(x, y);
        }
        return pts;
    }

  public:
    Geometry(pugi::xml_node geometry_node)
        : m_geometry{std::move(geometry_node)}
    {
    }

    std::optional<unsigned> group() const noexcept
    {
        auto g = m_geometry.attribute("group_id");
        if (g.empty())
            return std::nullopt;
        return g.as_uint();
    }

    std::string_view label() const noexcept
    {
        return m_geometry.attribute("label").as_string();
    }

    void draw_mask(cv::Mat &in_out) const noexcept
    {
        if (strcmp(m_geometry.name(), "polygon") == 0)
        {
            const std::vector<cv::Point> pts = parse_points(m_geometry);
            cv::fillPoly(in_out, pts, (unsigned char)255);
        }
        else if (strcmp(m_geometry.name(), "box") == 0)
        {
            const auto xtl = m_geometry.attribute("xtl").as_int();
            const auto ytl = m_geometry.attribute("ytl").as_int();
            const auto xbr = m_geometry.attribute("xbr").as_int();
            const auto ybr = m_geometry.attribute("ybr").as_int();

            cv::rectangle(in_out, cv::Rect{xtl, ytl, xbr - xtl, ybr - ytl}, 255,
                          cv::FILLED);
        }
        else if (strcmp(m_geometry.name(), "points") == 0)
        {
            const std::vector<cv::Point> pts = parse_points(m_geometry);

            for (auto &&p : pts)
            {
                cv::circle(in_out, p, 0, 255, cv::FILLED);
            }
        }
        else if (strcmp(m_geometry.name(), "polyline") == 0)
        {
            const std::vector<cv::Point> pts = parse_points(m_geometry);
            cv::polylines(in_out, pts, false, 255);
        }
        else if (strcmp(m_geometry.name(), "ellipse") == 0)
        {
            const auto x = m_geometry.attribute("cx").as_int();
            const auto y = m_geometry.attribute("cy").as_int();
            const auto rx = m_geometry.attribute("rx").as_int();
            const auto ry = m_geometry.attribute("ry").as_int();
            const auto rotation_grad =
                m_geometry.attribute("rotation").as_float(0.f);
            cv::ellipse(in_out, cv::Point(x, y), cv::Size(rx, ry),
                        rotation_grad, 0, 360, 255, cv::FILLED);
        }
    }
};

class Image
{
    pugi::xml_node m_image_node;

  public:
    Image(pugi::xml_node n) : m_image_node{std::move(n)} {}
    size_t width() const noexcept
    {
        if constexpr (sizeof(size_t) == sizeof(unsigned long long))
        {
            return m_image_node.attribute("width").as_ullong();
        }
        else if constexpr (sizeof(size_t) == sizeof(unsigned int))
        {
            return m_image_node.attribute("width").as_uint();
        }
        else
        {
            static_assert("Cannot determine size_t size");
        }
    }
    size_t height() const noexcept
    {
        if constexpr (sizeof(size_t) == sizeof(unsigned long long))
        {
            return m_image_node.attribute("height").as_ullong();
        }
        else if constexpr (sizeof(size_t) == sizeof(unsigned int))
        {
            return m_image_node.attribute("height").as_uint();
        }
        else
        {
            static_assert("Cannot determine size_t size");
        }
    }

    std::vector<std::string_view> labels() const
    {
        std::vector<std::string_view> result;
        for (Geometry geometry : m_image_node.children())
        {
            result.push_back(geometry.label());
        }
        return result;
    }

    cv::Mat empty_mask() const
    {
        const auto h = height();
        const auto w = width();
        return cv::Mat((int)h, (int)w, CV_8UC1, (unsigned char)0);
    }

    std::string_view filename() const noexcept
    {
        return m_image_node.attribute("name").as_string();
    }

    cv::Mat mask_combined(std::string_view label) const
    {
        cv::Mat result = empty_mask();

        for (Geometry geo : m_image_node.children())
        {
            if (geo.label() != label)
                continue;
            geo.draw_mask(result);
        }

        return result;
    }

    std::vector<cv::Mat> mask(std::string_view label) const
    {
        std::vector<cv::Mat> result;
        std::unordered_map<int, cv::Mat> groups;

        for (Geometry geo : m_image_node.children())
        {
            if (geo.label() != label)
                continue;

            auto group = geo.group();
            cv::Mat mat = empty_mask();
            if (group)
            {
                mat = groups.insert({group.value(), mat}).first->second;
            }
            geo.draw_mask(mat);
            result.push_back(mat);
        }

        return result;
    }

    std::unordered_map<std::string_view, cv::Mat> masks() const
    {
        std::unordered_map<std::string_view, cv::Mat> result;
        std::unordered_map<int, cv::Mat> groups;

        const auto h = height();
        const auto w = width();

        for (Geometry geometry : m_image_node.children())
        {
            auto label = geometry.label();
            auto mat = result
                           .try_emplace(label, (int)h, (int)w, CV_8UC1,
                                        (unsigned char)0)
                           .first->second;
            auto group = geometry.group();
            if (group)
            {
                mat = groups.try_emplace(group.value(), mat).first->second;
            }
            geometry.draw_mask(mat);
        }
        return result;
    }
};

class ImageIterator : public pugi::xml_named_node_iterator
{
  public:
    ImageIterator(const pugi::xml_named_node_iterator &it)
        : pugi::xml_named_node_iterator{it}
    {
    }

    using value_type = Image;
    using reference = Image;
    using pointer = Image;

    value_type operator*()
    {
        return Image{pugi::xml_named_node_iterator::operator*()};
    }
};

class CVATMaskGenerator
{
  public:
    static CVATMaskGenerator from_file(std::string_view file)
    {
        pugi::xml_document doc;
        doc.load_file(file.data());
        return CVATMaskGenerator(std::move(doc));
    }

    class ImageRange
    {
        std::pair<ImageIterator, ImageIterator> m_p;

      public:
        ImageRange(const ImageIterator &f, const ImageIterator &s) : m_p{f, s}
        {
        }
        ImageIterator begin() const { return m_p.first; }
        ImageIterator end() const { return m_p.second; }
    };

    ImageRange images() const
    {
        auto range = m_annotations.children("image");
        return ImageRange{range.begin(), range.end()};
    }

    CVATMaskGenerator(pugi::xml_document doc) : m_doc{std::move(doc)}
    {
        m_annotations = m_doc.child("annotations");
        m_task = m_annotations.child("meta").child("task");
    }

    std::vector<std::string_view> filenames() const
    {
        std::vector<std::string_view> files;
        for (const auto &image : m_annotations.children("image"))
        {
            const auto &attr = image.attribute("name");
            if (!attr.empty())
            {
                files.push_back(attr.as_string());
            }
        }
        return files;
    }

    std::vector<std::string_view> labels() const
    {
        std::vector<std::string_view> result;
        for (auto &&l : m_task.child("labels").children())
        {
            result.push_back(l.child("name").text().as_string());
        }
        return result;
    }

    std::vector<std::string_view> labels(std::string_view filename) const
    {
        std::vector<std::string_view> labels;
        for (const auto &image : m_annotations.children("image"))
        {
            if (filename != image.attribute("name").as_string())
            {
                continue;
            }
            for (const auto &geometry : image.children())
            {
                labels.push_back(geometry.attribute("label").as_string());
            }
        }
        return labels;
    }

    std::vector<cv::Mat> masks(std::string_view filename,
                               std::string_view label) const
    {
        std::vector<cv::Mat> mats;
        for (pugi::xml_node image : m_annotations.children("image"))
        {
            if (filename != image.attribute("name").as_string())
            {
                continue;
            }

            std::unordered_map<int, cv::Mat> group_ids;
            const auto width = image.attribute("width").as_int();
            const auto height = image.attribute("height").as_int();

            for (pugi::xml_node geometry : image.children())
            {
                if (label != geometry.attribute("label").as_string())
                {
                    continue;
                }

                const auto &group_id = geometry.attribute("group_id");
                cv::Mat mask;
                if (group_id.empty())
                {
                    mask = cv::Mat(height, width, CV_8UC1, (unsigned char)0);
                }
                else
                {
                    if (!group_ids.contains(group_id.as_int()))
                    {
                        group_ids.insert(
                            {group_id.as_int(), cv::Mat(height, width, CV_8UC1,
                                                        (unsigned char)0)});
                    }
                    mask = group_ids[group_id.as_int()];
                }

                if (strcmp(geometry.name(), "polygon") == 0)
                {
                    const std::vector<cv::Point> pts = parse_points(geometry);
                    cv::fillPoly(mask, pts, (unsigned char)255);
                }
                else if (strcmp(geometry.name(), "box") == 0)
                {
                    const auto xtl = geometry.attribute("xtl").as_int();
                    const auto ytl = geometry.attribute("ytl").as_int();
                    const auto xbr = geometry.attribute("xbr").as_int();
                    const auto ybr = geometry.attribute("ybr").as_int();

                    cv::rectangle(mask,
                                  cv::Rect{xtl, ytl, xbr - xtl, ybr - ytl}, 255,
                                  cv::FILLED);
                }
                else if (strcmp(geometry.name(), "points") == 0)
                {
                    const std::vector<cv::Point> pts = parse_points(geometry);

                    cv::Mat mat(height, width, CV_8UC1, (unsigned char)0);
                    for (const auto &p : pts)
                    {
                        cv::circle(mask, p, 0, 255, cv::FILLED);
                    }
                }
                else if (strcmp(geometry.name(), "polyline") == 0)
                {
                    const std::vector<cv::Point> pts = parse_points(geometry);
                    cv::polylines(mask, pts, false, 255);
                }
                else if (strcmp(geometry.name(), "ellipse") == 0)
                {
                    const auto x = geometry.attribute("cx").as_int();
                    const auto y = geometry.attribute("cy").as_int();
                    const auto rx = geometry.attribute("rx").as_int();
                    const auto ry = geometry.attribute("ry").as_int();
                    const auto rotation_grad =
                        geometry.attribute("rotation").as_float(0.f);
                    cv::ellipse(mask, cv::Point(x, y), cv::Size(rx, ry),
                                rotation_grad, 0, 360, 255, cv::FILLED);
                }
                mats.push_back(mask);
            }
        }
        return mats;
    }

  private:
    static std::vector<cv::Point>
    parse_points(const pugi::xml_node &node_with_point_attr)
    {
        const auto ptr_start =
            node_with_point_attr.attribute("points").as_string();
        const auto ptr_end = ptr_start + strlen(ptr_start);
        auto cur = ptr_start;
        std::vector<cv::Point> pts;
        while (cur < ptr_end)
        {
            auto x_coord_end = strchr(cur, ',');
            if (x_coord_end == nullptr)
                x_coord_end = ptr_end;
            auto y_coord_end = strchr(cur, ';');
            y_coord_end = (y_coord_end != nullptr) ? y_coord_end : ptr_end;

            int x, y;
            std::from_chars(cur, x_coord_end, x);
            std::from_chars(x_coord_end + 1,
                            (y_coord_end == nullptr) ? ptr_end : y_coord_end,
                            y);
            cur = y_coord_end + 1;
            pts.emplace_back(x, y);
        }
        return pts;
    }

    pugi::xml_document m_doc;
    pugi::xml_node m_annotations;
    pugi::xml_node m_task;
};

void write_masks_to_directory(std::string_view xml_file,
                              std::filesystem::path output_directory)
{
    auto &&generator = CVATMaskGenerator::from_file(xml_file);
    auto &&labels = generator.labels();

    for (auto &&l : labels)
    {
        auto &&label_dir = output_directory / l;

        if (!std::filesystem::exists(label_dir) ||
            !std::filesystem::is_directory(label_dir))
        {
            std::filesystem::create_directories(label_dir);
        }
    }

    std::vector<std::future<void>> futures;

    for (auto &&image : generator.images())
    {
        futures.push_back(std::async(
            std::launch::async,
            [image, &output_directory, &labels]()
            {
                auto &&filename = std::filesystem::path(image.filename());
                filename = filename.replace_extension(".png");

                for (auto &&l : labels)
                {
                    auto mat = image.mask_combined(l);
                    cv::imwrite((output_directory / l / filename).string(),
                                mat);
                }
            }));
    }

    for (auto &&f : futures)
    {
        f.wait();
    }
}

int main(int argc, char **argv)
{
    CLI::App app{"CVAT Mask generator\nhttps://github.com/TinyTinni/CVATTools"};

    std::string cvat_file = "annoations.xml";
    app.add_option("CVAT XML", cvat_file, "CVAT XML file")
        ->check(CLI::ExistingFile)
        ->required();
    std::string output_directory = "./";
    app.add_option("OUTDIR", output_directory, "Output directory")->required();

    auto start = std::chrono::high_resolution_clock::now();

    CLI11_PARSE(app, argc, argv);

    try
    {
        write_masks_to_directory(cvat_file, output_directory);
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what();
        return 1;
    }

    std::cout << "processing time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::high_resolution_clock::now() - start)
                     .count()
              << "ms\n";

    return 0;
}
